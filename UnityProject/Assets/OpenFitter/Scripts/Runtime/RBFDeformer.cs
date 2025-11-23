// ----------------------------------------------------------------------------
// Copyright (C) [2025] tallcat
//
// This file is free software: you can redistribute it and/or modify
// it under the terms of the GNU General Public License as published by
// the Free Software Foundation, either version 3 of the License, or
// (at your option) any later version.
//
// This file is distributed in the hope that it will be useful,
// but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
// See the accompanying LICENSE file for more details.
// ----------------------------------------------------------------------------

using System.Collections.Generic;
using UnityEngine;
using Unity.Collections;
using Unity.Jobs;
using Unity.Mathematics;
using Unity.Burst;
using System.IO;
using Newtonsoft.Json;

[System.Serializable]
public class RBFData
{
    public float epsilon;
    public List<List<float>> centers;
    public List<List<float>> weights;
    public List<List<float>> poly_weights;
}

[ExecuteInEditMode] // エディタ上で動作することを明示
public class RBFDeformer : MonoBehaviour
{
    public string jsonFilePath = "rbf_data.json";

    // エディタ拡張からアクセスできるようにpublic/SerializeFieldに変更
    [SerializeField] private Mesh originalMesh;
    [SerializeField] private Mesh deformedMesh;
    
    // ゲッター（エディタ拡張用）
    public Mesh OriginalMesh => originalMesh;
    public Mesh DeformedMesh => deformedMesh;

    // Job用データ
    private NativeArray<float3> originalVertices;
    private NativeArray<float3> deformedVertices;
    private NativeArray<float3> centers;
    private NativeArray<float3> weights;
    private NativeArray<float3> polyWeights;
    
    private float epsilon;

    // コンポーネント削除時やスクリプト再コンパイル時にメモリを解放
    void OnDisable()
    {
        DisposeNativeArrays();
    }

    void OnDestroy()
    {
        DisposeNativeArrays();
    }

    public void DisposeNativeArrays()
    {
        if (originalVertices.IsCreated) originalVertices.Dispose();
        if (deformedVertices.IsCreated) deformedVertices.Dispose();
        if (centers.IsCreated) centers.Dispose();
        if (weights.IsCreated) weights.Dispose();
        if (polyWeights.IsCreated) polyWeights.Dispose();
    }

    // エディタから「実行」ボタンで呼ばれる一括処理関数
    public void RunDeformationInEditor()
    {
        // 1. メッシュの準備
        if (!InitMesh()) return;

        // 2. データのロード
        if (!LoadRBFData()) return;

        // 3. 計算と適用
        ApplyRBF();
    }

    bool InitMesh()
    {
        var mf = GetComponent<MeshFilter>();
        var smr = GetComponent<SkinnedMeshRenderer>();

        if (mf == null && smr == null)
        {
            Debug.LogError("MeshFilter or SkinnedMeshRenderer is missing.");
            return false;
        }

        // オリジナルメッシュの取得
        if (smr != null) originalMesh = smr.sharedMesh;
        else originalMesh = mf.sharedMesh;

        if (originalMesh == null)
        {
            Debug.LogError("Original mesh is missing.");
            return false;
        }

        // プレビュー用メッシュの作成
        // 以前のプレビューメッシュがあれば破棄（メモリリーク防止）
        if (deformedMesh != null)
        {
            // シーンに残らないよう即時破棄
            if (Application.isPlaying) Destroy(deformedMesh);
            else DestroyImmediate(deformedMesh);
        }

        deformedMesh = Instantiate(originalMesh);
        deformedMesh.name = originalMesh.name + "_Preview";
        // シーン保存時にこのメッシュを含めない（Assetとして保存するまで）
        deformedMesh.hideFlags = HideFlags.DontSaveInEditor | HideFlags.DontSaveInBuild;
        
        if (smr != null) smr.sharedMesh = deformedMesh;
        else mf.mesh = deformedMesh;

        return true;
    }

    bool LoadRBFData()
    {
        string path = Path.Combine(Application.dataPath, jsonFilePath);
        if (!File.Exists(path))
        {
            Debug.LogError("JSON file not found: " + path);
            return false;
        }

        try 
        {
            string jsonStr = File.ReadAllText(path);
            var data = JsonConvert.DeserializeObject<RBFData>(jsonStr);

            this.epsilon = data.epsilon;
            
            DisposeNativeArrays(); // 安全のためリセット

            // 軸変換: Blender (Right-Handed Z-Up) -> Unity (Left-Handed Y-Up)
            // Mapping: (-x, z, -y)
            // これはBoneDeformer.csの実装と一致させるための変更です。
            var centersArr = ConvertToUnitySpace(data.centers);
            var weightsArr = ConvertToUnitySpace(data.weights);
            var polyArr = ConvertToUnitySpace(data.poly_weights);

            // 多項式項の入力座標系の補正
            // Poly = Bias + C_x * x_in + C_y * y_in + C_z * z_in
            // Unity入力 (x_u, y_u, z_u) に対して:
            // x_in_blender = -x_u
            // y_in_blender = -z_u
            // z_in_blender = y_u
            
            // Row 0 (Bias): 変換済み (ConvertToUnitySpaceで出力座標系は変換されている)
            // Row 1 (X coeff): x_in = -x_u なので、係数を反転
            polyArr[1] = -polyArr[1];
            
            // Row 2 (Y coeff) & Row 3 (Z coeff):
            // Term Y: C_y * y_in = C_y * (-z_u) -> UnityのZ係数(Row 3)に -C_y をセット
            // Term Z: C_z * z_in = C_z * (y_u)  -> UnityのY係数(Row 2)に C_z をセット
            
            float3 oldRow2 = polyArr[2]; // C_y (converted to Unity output space)
            float3 oldRow3 = polyArr[3]; // C_z (converted to Unity output space)
            
            polyArr[2] = oldRow3;  // New Y coeff = Old Z coeff
            polyArr[3] = -oldRow2; // New Z coeff = -Old Y coeff

            centers = new NativeArray<float3>(centersArr, Allocator.Persistent);
            weights = new NativeArray<float3>(weightsArr, Allocator.Persistent);
            polyWeights = new NativeArray<float3>(polyArr, Allocator.Persistent);

            return true;
        }
        catch (System.Exception e)
        {
            Debug.LogError($"JSON Load Error: {e.Message}");
            return false;
        }
    }

    float3[] ConvertToUnitySpace(List<List<float>> list)
    {
        float3[] result = new float3[list.Count];
        for (int i = 0; i < list.Count; i++)
        {
            // Blender (x, y, z) -> Unity (-x, z, -y)
            result[i] = new float3(-list[i][0], list[i][2], -list[i][1]);
        }
        return result;
    }

    void ApplyRBF()
    {
        Vector3[] meshVerts = originalMesh.vertices;

        // NativeArrayの再確保
        if (!originalVertices.IsCreated || originalVertices.Length != meshVerts.Length)
        {
            if (originalVertices.IsCreated) originalVertices.Dispose();
            if (deformedVertices.IsCreated) deformedVertices.Dispose();
            
            originalVertices = new NativeArray<float3>(meshVerts.Length, Allocator.Persistent);
            deformedVertices = new NativeArray<float3>(meshVerts.Length, Allocator.Persistent);
        }

        // データのコピー
        // エディタ実行なのでReinterpretは使わず安全にコピー
        for(int i=0; i<meshVerts.Length; i++) originalVertices[i] = meshVerts[i];

        var job = new RBFDeformJob
        {
            vertices = originalVertices,
            deformedVertices = deformedVertices,
            centers = centers,
            weights = weights,
            polyWeights = polyWeights,
            epsilon = epsilon,
            localToWorld = transform.localToWorldMatrix,
            inverseRotation = Quaternion.Inverse(transform.rotation)
        };

        // 実行と待機
        job.Schedule(originalVertices.Length, 64).Complete();

        // 結果の書き戻し
        Vector3[] resultVerts = new Vector3[meshVerts.Length];
        for(int i=0; i<meshVerts.Length; i++) resultVerts[i] = deformedVertices[i];

        deformedMesh.vertices = resultVerts;
        deformedMesh.RecalculateNormals();
        deformedMesh.RecalculateBounds();
        
        Debug.Log($"<color=cyan>[RBF Deformer]</color> Applied successfully. ({meshVerts.Length} vertices)");
    }

    [BurstCompile]
    struct RBFDeformJob : IJobParallelFor
    {
        [ReadOnly] public NativeArray<float3> vertices;
        [ReadOnly] public NativeArray<float3> centers;
        [ReadOnly] public NativeArray<float3> weights;
        [ReadOnly] public NativeArray<float3> polyWeights;
        [ReadOnly] public float epsilon;
        [ReadOnly] public float4x4 localToWorld;
        [ReadOnly] public quaternion inverseRotation;

        [WriteOnly] public NativeArray<float3> deformedVertices;

        public void Execute(int i)
        {
            float3 p_local = vertices[i];
            float3 p_world = math.transform(localToWorld, p_local);
            float3 displacement = float3.zero;
            float eps2 = epsilon * epsilon;

            for (int j = 0; j < centers.Length; j++)
            {
                float distSq = math.distancesq(p_world, centers[j]);
                float phi = math.sqrt(distSq + eps2);
                displacement += weights[j] * phi;
            }

            displacement += polyWeights[0];
            displacement += polyWeights[1] * p_world.x;
            displacement += polyWeights[2] * p_world.y;
            displacement += polyWeights[3] * p_world.z;

            float3 disp_local = math.rotate(inverseRotation, displacement);
            deformedVertices[i] = p_local + disp_local;
        }
    }
}
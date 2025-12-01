# OpenFitter

[English](#english) | [日本語](#japanese)

<a name="english"></a>

An open-source avatar clothing fitting tool implementation project designed to handle data formats from the "MochiFitter" workflow.
Based on the GPL-3 components released by Nine Gates, OpenFitter provides a pipeline for transferring clothing between different avatar shapes using RBF (Radial Basis Function) deformation and Bone Pose transfer.

***

**Status**: Core Refactoring

The project is currently in the phase of analyzing and structuring the original code (Upstream) to ensure complete data compatibility and reproducibility with the original product.

***

## Current Roadmap: Architecture Refresh

We are transitioning from our initial implementation to the core logic obtained from upstream to achieve more accurate behavior.

*   Legacy Implementation:
    The bone and meshes control logic originally inferred and implemented independently. This corresponds to the C# code and other components currently in the repository.

*   Upstream-based Implementation (In Progress):
    GPL code derived from the original product, located in `src/upstream/`.
    To achieve full compatibility, it is necessary to decipher and refactor this massive codebase spanning approximately 20,000 lines, then re-implement it.
    The repository is currently prioritizing the analysis and refactoring of this large-scale codebase.

## Features (Legacy)

*   Blender Tools:
    *   Bone Pose Exporter: Exports the difference between two armatures (Source → Target) as a JSON file.
    *   RBF Field Exporter: Calculates the deformation field between a "Basis" shape key and a "Target" shape key and exports it as RBF data (JSON).
*   Unity Tools:
    *   OpenFitter Converter: A standalone Editor Window to convert clothing prefabs.
        *   Applies RBF deformation to meshes.
        *   Applies Bone Pose transformation to the armature.
        *   Asset Saving: Automatically saves deformed meshes and creates a ready-to-use Prefab.

## Installation

### Blender Addon
1.  Copy the `blender_addon` folder (or zip it) and install it in Blender via `Edit > Preferences > Add-ons`.
2.  Enable "Import-Export: OpenFitter Tools".
3.  Access the tools via the **Sidebar (N-Panel) > OpenFitter** tab.

### Unity Package
1.  Copy the `UnityProject/Assets/OpenFitter` folder into your Unity project's `Assets` folder.
2.  Ensure you have the `Newtonsoft Json` package installed.

## Acknowledgements & Compliance

This software is an independent open-source implementation interoperable with "MochiFitter" data.

* Core Engine: Based on the GPL-3 source code (Upstream) being released by Nine Gates.
* UI & Frontend: Being developed independently for this project.

Development Policy and Compliance:
To ensure compliance with the original product's EULA (specifically regarding reverse engineering prohibitions), this project is being developed without any reverse engineering (decompilation, disassembly, etc.) of proprietary binaries. All independent implementations are referencing only publicly available documentation and GPL-3 source code.

This is an unofficial project and has not received any approval or endorsement from Nine Gates or "MochiFitter".

## License

This project is licensed under the GNU General Public License v3.0 (GPL-3).
See the LICENSE file for details.

---

<a name="japanese"></a>

# OpenFitter (日本語)

「もちふぃった～」ワークフローのデータ形式に対応した、オープンソースのアバター衣装フィッティングツール実装プロジェクトです。
Nine Gatesによって公開されたGPL-3コンポーネントに基づき、RBF変形とボーンポーズ転送のパイプラインを提供します。

***

**ステータス**: Core Refactoring

現在、本プロジェクトは本家製品との完全なデータ互換性と再現性を確保するため、オリジナルのコード（Upstream）の解析および構造化フェーズにあります。

***

## アーキテクチャの刷新について

当初の独自実装から、より正確な挙動を実現するために、upstreamより取得したコアロジックへの移行を進めています。

*   レガシー実装:
    これまで独自に推測・実装していたボーン・メッシュの制御ロジック。現在のリポジトリに含まれるC#コード等が該当します。

*   Upstreamに基づく実装 (進行中):
    `src/upstream/` に追加された、本家製品由来のGPLコード。
    完全な互換動作を実現するために、約20,000行に及ぶコードを解明し、リファクタリングして再実装するプロセスを進めています。
    本リポジトリは現在、この大規模なコードベースの解析とリファクタリングを課題としています。

## 機能(レガシー)

*   Blenderツール:
    *   Bone Pose Exporter: 2つのアーマチュア間(ソース→ターゲット)の差分をJSONファイルとしてエクスポートします。
    *   RBF Field Exporter: "Basis"(基準)シェイプキーと "Target"(変形後)シェイプキーの間の変形フィールドを計算し、RBFデータ(JSON)としてエクスポートします。
*   Unityツール:
    *   OpenFitter Converter: 衣装プレハブを変換するための独立したエディタウィンドウです。
        *   メッシュへのRBF変形の適用
        *   アーマチュアへのボーンポーズ変形の適用
        *   アセット保存: 変形されたメッシュを自動的に保存し、すぐに使用可能なプレハブを作成します。

## インストール

### Blenderアドオン
1.  `blender_addon` フォルダをコピー(またはzip圧縮)し、Blenderの `Edit > Preferences > Add-ons` からインストールします。
2.  "Import-Export: OpenFitter Tools" を有効にします。
3.  **Sidebar (N-Panel) > OpenFitter** タブからツールにアクセスします。

### Unityパッケージ
1.  `UnityProject/Assets/OpenFitter` フォルダを、Unityプロジェクトの `Assets` フォルダ内にコピーします。
2.  `Newtonsoft Json` パッケージがインストールされていることを確認してください。

## クレジット・規約の遵守

本ソフトウェアは「もちふぃった～」のデータと相互運用性を持つ、独立したオープンソース実装です。

* コアエンジン: Nine Gatesにより公開されているGPL-3ソースコード(Upstream)を基にしています。
* UI・フロントエンド: 本プロジェクトのために独自に実装しているものです。

開発方針と規約の遵守について:
オリジナル製品の利用規約(特にリバースエンジニアリングの禁止に関する条項)を遵守するため、本プロジェクトはプロプライエタリなバイナリに対するリバースエンジニアリング(逆コンパイル、逆アセンブル等)を一切行わずに開発しています。独自実装部分は、公開されているドキュメントおよびGPL-3ソースコードのみを参照しています。

本プロジェクトは非公式なものであり、Nine Gatesおよび「もちふぃった～」からのあらゆる承認、認可等も受けていません。

## ライセンス

本プロジェクトは GNU General Public License v3.0 (GPL-3) の下で公開されています。
詳細は LICENSE ファイルをご確認ください。
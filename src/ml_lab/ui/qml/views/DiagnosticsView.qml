import QtQuick
import QtQuick.Controls
import QtQuick.Dialogs
import QtQuick.Layouts
import "../components"

Item {
    FileDialog { id: exportDialog; title: "Export diagnostics bundle"; fileMode: FileDialog.SaveFile; nameFilters: ["ZIP archive (*.zip)"]; defaultSuffix: "zip"; onAccepted: appController.exportDiagnostics(selectedFile.toString()) }
    ColumnLayout {
        anchors.fill: parent; spacing: 16
        RowLayout {
            Layout.fillWidth: true
            ColumnLayout { Text { text: "Diagnostics"; color: Theme.text; font.pixelSize: 26; font.weight: Font.Bold }; Text { text: "Lightweight host discovery — no Torch import required."; color: Theme.muted; font.pixelSize: 13 } }
            Item { Layout.fillWidth: true }; LabButton { text: "Open logs"; onClicked: appController.openLogsFolder() }; LabButton { text: "Export bundle"; onClicked: exportDialog.open() }; LabButton { text: "Refresh"; primary: true; onClicked: appController.refreshDiagnostics() }
        }
        GridLayout {
            columns: width > 900 ? 3 : 2; columnSpacing: 12; rowSpacing: 12; Layout.fillWidth: true
            Repeater {
                model: [["Operating system", appController.diagnostics.os || "—"], ["Processor", appController.diagnostics.cpu || "—"], ["Logical CPUs", appController.diagnostics.logicalCpus || "—"], ["System memory", appController.diagnostics.ram || "—"], ["Workspace free", appController.diagnostics.diskFree || "—"], ["NVIDIA GPU", appController.diagnostics.gpu || "—"], ["NVIDIA VRAM", appController.diagnostics.vram || "—"], ["Adapters", appController.diagnostics.adapterCount || "0"], ["Runtime packs", appController.diagnostics.runtimeCount || "0"]]
                delegate: Panel { required property var modelData; Layout.fillWidth: true; implicitHeight: 96; Column { anchors.fill: parent; anchors.margins: 16; spacing: 8; Text { text: modelData[0]; color: Theme.dim; font.pixelSize: 12 }; Text { text: modelData[1]; color: Theme.text2; font.pixelSize: 15; font.weight: Font.DemiBold; elide: Text.ElideRight; width: parent.width } } }
            }
        }
        Panel { Layout.fillWidth: true; implicitHeight: 72; visible: (appController.diagnostics.extensionErrors || 0) > 0; Text { anchors.fill: parent; anchors.margins: 16; text: appController.diagnostics.extensionErrors + " extension manifest(s) were rejected. Export diagnostics for details."; color: Theme.badText; wrapMode: Text.WordWrap; verticalAlignment: Text.AlignVCenter } }
        Item { Layout.fillHeight: true }
    }
}

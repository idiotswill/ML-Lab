import QtQuick
import QtQuick.Controls
import QtQuick.Layouts
import "../components"

Item {
    ColumnLayout {
        anchors.fill: parent; spacing: 16
        ColumnLayout { Text { text: "Settings"; color: Theme.text; font.pixelSize: 26; font.weight: Font.Bold }; Text { text: "Lab presentation and local application preferences."; color: Theme.muted; font.pixelSize: 13 } }
        Panel {
            Layout.fillWidth: true; implicitHeight: 132
            RowLayout {
                anchors.fill: parent; anchors.margins: 18; spacing: 16
                ColumnLayout { Layout.fillWidth: true; Text { text: "Appearance"; color: Theme.text2; font.pixelSize: 16; font.weight: Font.DemiBold }; Text { text: "Follow Windows, or override the Lab theme."; color: Theme.muted; font.pixelSize: 12 } }
                ComboBox { id: themePicker; model: ["System", "Dark", "Light"]; currentIndex: appController.themeMode === "dark" ? 1 : appController.themeMode === "light" ? 2 : 0; onActivated: appController.setThemeMode(["system", "dark", "light"][currentIndex]) }
            }
        }
        Panel {
            Layout.fillWidth: true; implicitHeight: 110
            ColumnLayout { anchors.fill: parent; anchors.margins: 18; spacing: 7; Text { text: "Authority boundary"; color: Theme.text2; font.pixelSize: 16; font.weight: Font.DemiBold }; Text { text: "This Lab can train, score, compare and package models. It does not own Frankenhomie game state or integration approval."; color: Theme.muted; wrapMode: Text.WordWrap; Layout.fillWidth: true }; StatusPill { text: "INTEGRATION NO-GO"; tone: "warn" } }
        }
        Item { Layout.fillHeight: true }
    }
}

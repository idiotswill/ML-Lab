import QtQuick
import QtQuick.Controls
import QtQuick.Dialogs
import QtQuick.Layouts
import "../components"

Item {
    id: root

    FolderDialog {
        id: createFolder
        title: "Choose a folder for your ML Lab workspace"
        onAccepted: appController.openWorkspace(selectedFolder.toString(), true)
    }

    FolderDialog {
        id: openFolder
        title: "Open an existing ML Lab workspace"
        onAccepted: appController.openWorkspace(selectedFolder.toString(), false)
    }

    ColumnLayout {
        anchors.centerIn: parent
        width: Math.min(680, root.width - 64)
        spacing: 18

        Text {
            text: "Frankenhomie ML Lab"
            color: Theme.text
            font.pixelSize: 34
            font.weight: Font.Bold
            Layout.alignment: Qt.AlignHCenter
        }

        Text {
            text: "A local Windows workbench for building, comparing and breaking bounded ML projects — without becoming another game authority."
            color: Theme.muted
            wrapMode: Text.WordWrap
            horizontalAlignment: Text.AlignHCenter
            font.pixelSize: 15
            lineHeight: 1.35
            Layout.fillWidth: true
        }

        Panel {
            Layout.fillWidth: true
            implicitHeight: cardContent.implicitHeight + 44

            ColumnLayout {
                id: cardContent
                anchors.fill: parent
                anchors.margins: 22
                spacing: 13

                Text {
                    text: "Start with a workspace"
                    color: Theme.text2
                    font.pixelSize: 18
                    font.weight: Font.DemiBold
                }

                Text {
                    text: "The workspace stores Lab metadata, immutable experiment artifacts and logs. It never becomes Frankenhomie's campaign database."
                    color: Theme.muted
                    wrapMode: Text.WordWrap
                    Layout.fillWidth: true
                }

                RowLayout {
                    spacing: 10

                    LabButton {
                        id: createWorkspaceButton
                        text: "Create workspace"
                        primary: true
                        focus: true
                        onClicked: createFolder.open()
                    }

                    LabButton {
                        text: "Open workspace"
                        onClicked: openFolder.open()
                    }
                }
            }
        }
    }

    Component.onCompleted: createWorkspaceButton.forceActiveFocus()
}

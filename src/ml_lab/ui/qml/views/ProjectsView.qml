import QtQuick
import QtQuick.Controls
import QtQuick.Layouts
import "../components"

Item {
    id: root

    ContractSnapshotsDialog {
        id: contractDialog
    }

    Dialog {
        id: archiveDialog
        modal: true
        anchors.centerIn: parent
        width: 480
        title: "Archive project?"
        standardButtons: Dialog.NoButton
        property string projectId: ""
        property string projectName: ""

        background: Rectangle {
            color: Theme.surface
            radius: 12
            border.color: Theme.warnBorder
        }

        contentItem: ColumnLayout {
            spacing: 14

            Text {
                Layout.fillWidth: true
                text: "Archive “" + archiveDialog.projectName + "”?"
                color: Theme.text
                font.pixelSize: 16
                font.weight: Font.DemiBold
                wrapMode: Text.WordWrap
            }

            Text {
                Layout.fillWidth: true
                text: "Scope: this hides the project from the active project list. Immutable datasets, experiments, failures, models, bundles, hashes and receipts are preserved and are not deleted."
                color: Theme.muted
                font.pixelSize: 12
                wrapMode: Text.WordWrap
            }

            RowLayout {
                Layout.fillWidth: true
                Item { Layout.fillWidth: true }

                LabButton {
                    text: "Keep project"
                    onClicked: archiveDialog.close()
                }

                LabButton {
                    text: "Archive project"
                    primary: true
                    onClicked: {
                        appController.archiveProject(archiveDialog.projectId)
                        archiveDialog.close()
                    }
                }
            }
        }
    }

    Dialog {
        id: createDialog
        modal: true
        anchors.centerIn: parent
        width: 500
        title: "New ML project"
        standardButtons: Dialog.NoButton
        onOpened: projectName.forceActiveFocus()

        background: Rectangle {
            color: Theme.surface
            radius: 12
            border.color: Theme.borderStrong
        }

        contentItem: ColumnLayout {
            spacing: 12

            Text {
                text: "Project name"
                color: Theme.muted
            }

            TextField {
                id: projectName
                Layout.fillWidth: true
                placeholderText: "Phase A Semantic Intake"
                activeFocusOnTab: true
                Accessible.name: "Project name"
            }

            Text {
                text: "Project adapter"
                color: Theme.muted
            }

            ComboBox {
                id: adapterPicker
                Layout.fillWidth: true
                model: appController.adapters
                textRole: "name"
                activeFocusOnTab: true
                Accessible.name: "Project adapter"
            }

            Text {
                text: "Description"
                color: Theme.muted
            }

            TextArea {
                id: description
                Layout.fillWidth: true
                implicitHeight: 90
                wrapMode: TextArea.Wrap
                activeFocusOnTab: true
                Accessible.name: "Project description"
            }

            RowLayout {
                Layout.alignment: Qt.AlignRight

                LabButton {
                    text: "Cancel"
                    onClicked: createDialog.close()
                }

                LabButton {
                    text: "Create project"
                    primary: true
                    enabled: projectName.text.trim().length > 0 && adapterPicker.currentIndex >= 0
                    onClicked: {
                        const adapter = appController.adapters[adapterPicker.currentIndex]
                        appController.createProject(projectName.text, adapter.id, description.text)
                        createDialog.close()
                        projectName.text = ""
                        description.text = ""
                    }
                }
            }
        }
    }

    ColumnLayout {
        anchors.fill: parent
        spacing: 16

        RowLayout {
            Layout.fillWidth: true

            ColumnLayout {
                spacing: 2

                Text {
                    text: "Projects"
                    color: Theme.text
                    font.pixelSize: 26
                    font.weight: Font.Bold
                }

                Text {
                    text: "Independent ML problems share the Lab infrastructure, not their authority."
                    color: Theme.muted
                    font.pixelSize: 13
                }
            }

            Item {
                Layout.fillWidth: true
            }

            LabButton {
                text: "New project"
                primary: true
                onClicked: createDialog.open()
            }
        }

        RowLayout {
            Layout.fillWidth: true
            Layout.fillHeight: true
            spacing: 12

            Panel {
                Layout.fillWidth: true
                Layout.fillHeight: true
                Layout.preferredWidth: 680

                ListView {
                    id: list
                    anchors.fill: parent
                    anchors.margins: 10
                    clip: true
                    spacing: 7
                    model: appController.projects

                    delegate: ItemDelegate {
                        id: projectDelegate
                        required property var modelData

                        width: list.width
                        height: 82
                        leftPadding: 14
                        rightPadding: 14
                        topPadding: 10
                        bottomPadding: 10
                        activeFocusOnTab: true
                        Accessible.role: Accessible.Button
                        Accessible.name: modelData.name
                        onClicked: appController.openProject(modelData.id)

                        contentItem: RowLayout {
                            spacing: 12

                            ColumnLayout {
                                Layout.fillWidth: true
                                spacing: 5

                                Text {
                                    text: modelData.name
                                    color: Theme.text2
                                    font.pixelSize: 16
                                    font.weight: Font.DemiBold
                                }

                                Text {
                                    text: modelData.description || "No description"
                                    color: Theme.muted
                                    elide: Text.ElideRight
                                    Layout.fillWidth: true
                                }
                            }

                            StatusPill {
                                text: modelData.adapter_id
                                tone: "neutral"
                            }

                            StatusPill {
                                text: "NO-GO"
                                tone: "warn"
                            }
                        }

                        background: Rectangle {
                            radius: 9
                            color: appController.selectedProject.id === modelData.id
                                   ? Theme.accentSurface
                                   : (projectDelegate.hovered ? Theme.hover : Theme.surfaceAlt)
                            border.color: projectDelegate.activeFocus ? Theme.accent
                                          : (appController.selectedProject.id === modelData.id
                                             ? Theme.accentBorder : Theme.border)
                            border.width: projectDelegate.activeFocus ? 2 : 1
                        }
                    }

                    footer: Item {
                        width: list.width
                        height: appController.projects.length === 0 ? 250 : 0

                        ColumnLayout {
                            anchors.centerIn: parent
                            visible: appController.projects.length === 0
                            width: Math.min(parent.width - 40, 520)
                            spacing: 8

                            Text {
                                Layout.alignment: Qt.AlignHCenter
                                text: "Your first Lab project starts here"
                                color: Theme.text
                                font.pixelSize: 18
                                font.weight: Font.DemiBold
                            }

                            Text {
                                Layout.fillWidth: true
                                text: "1. Create a bounded project and choose its adapter.\n2. Freeze or import data in Data Studio (Ctrl+2).\n3. Launch an experiment (Ctrl+3), compare results (Ctrl+4), preserve failures (Ctrl+5), then package only a RELEASE_CANDIDATE (Ctrl+7)."
                                color: Theme.muted
                                font.pixelSize: 12
                                wrapMode: Text.WordWrap
                                horizontalAlignment: Text.AlignHCenter
                            }

                            Text {
                                Layout.fillWidth: true
                                text: "Nothing here installs into Frankenhomie. Integration remains NO-GO."
                                color: Theme.warnText
                                font.pixelSize: 11
                                wrapMode: Text.WordWrap
                                horizontalAlignment: Text.AlignHCenter
                            }
                        }
                    }
                }
            }

            Panel {
                Layout.preferredWidth: 310
                Layout.fillHeight: true
                visible: Object.keys(appController.selectedProject).length > 0

                ColumnLayout {
                    anchors.fill: parent
                    anchors.margins: 18
                    spacing: 12

                    Text {
                        text: "PROJECT"
                        color: Theme.accent
                        font.pixelSize: 11
                        font.letterSpacing: 1.4
                        font.weight: Font.Bold
                    }

                    Text {
                        text: appController.selectedProject.name || ""
                        color: Theme.text
                        font.pixelSize: 20
                        font.weight: Font.Bold
                        wrapMode: Text.WordWrap
                        Layout.fillWidth: true
                    }

                    Text {
                        text: appController.selectedProject.description || "No description"
                        color: Theme.muted
                        wrapMode: Text.WordWrap
                        Layout.fillWidth: true
                    }

                    Rectangle {
                        Layout.fillWidth: true
                        height: 1
                        color: Theme.border
                    }

                    Text {
                        text: "Adapter"
                        color: Theme.dim
                        font.pixelSize: 11
                    }

                    Text {
                        text: appController.selectedProject.adapter_id || ""
                        color: Theme.text2
                        font.pixelSize: 13
                        wrapMode: Text.WrapAnywhere
                        Layout.fillWidth: true
                    }

                    Text {
                        text: "Contract snapshots"
                        color: Theme.dim
                        font.pixelSize: 11
                        Layout.topMargin: 4
                    }

                    RowLayout {
                        Layout.fillWidth: true

                        Text {
                            Layout.fillWidth: true
                            text: String(appController.modelsRegistry.contractSnapshots.snapshotRows.length) +
                                  " frozen"
                            color: Theme.text2
                            font.pixelSize: 12
                        }

                        StatusPill {
                            text: appController.modelsRegistry.contractSnapshots.captureSupported ?
                                  "DECLARED" : "UNDECLARED"
                            tone: appController.modelsRegistry.contractSnapshots.captureSupported ?
                                  "good" : "neutral"
                        }
                    }

                    LabButton {
                        text: "Contract snapshots"
                        Layout.fillWidth: true
                        onClicked: contractDialog.open()
                    }

                    Text {
                        text: "Integration gate"
                        color: Theme.dim
                        font.pixelSize: 11
                        Layout.topMargin: 4
                    }

                    StatusPill {
                        text: "NO-GO"
                        tone: "warn"
                    }

                    Item {
                        Layout.fillHeight: true
                    }

                    Text {
                        text: "Archiving hides the project but does not delete immutable history."
                        color: Theme.dim
                        font.pixelSize: 11
                        wrapMode: Text.WordWrap
                        Layout.fillWidth: true
                    }

                    LabButton {
                        text: "Archive project"
                        Layout.fillWidth: true
                        onClicked: {
                            archiveDialog.projectId = String(appController.selectedProject.id)
                            archiveDialog.projectName = String(appController.selectedProject.name)
                            archiveDialog.open()
                        }
                    }
                }
            }
        }
    }
}

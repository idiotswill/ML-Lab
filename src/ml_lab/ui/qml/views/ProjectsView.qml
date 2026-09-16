import QtQuick
import QtQuick.Controls
import QtQuick.Layouts
import "../components"

Item {
    id: root

    Dialog {
        id: createDialog
        modal: true
        anchors.centerIn: parent
        width: 500
        title: "New ML project"
        standardButtons: Dialog.NoButton

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

                    delegate: Rectangle {
                        required property var modelData

                        width: list.width
                        height: 82
                        radius: 9
                        color: appController.selectedProject.id === modelData.id
                               ? Theme.accentSurface
                               : (hover.hovered ? Theme.hover : Theme.surfaceAlt)
                        border.color: appController.selectedProject.id === modelData.id
                                      ? Theme.accentBorder : Theme.border

                        HoverHandler {
                            id: hover
                        }

                        TapHandler {
                            onTapped: appController.openProject(modelData.id)
                        }

                        RowLayout {
                            anchors.fill: parent
                            anchors.margins: 14
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
                    }

                    footer: Item {
                        width: list.width
                        height: appController.projects.length === 0 ? 180 : 0

                        Text {
                            anchors.centerIn: parent
                            visible: appController.projects.length === 0
                            text: "No projects yet. Create one without committing to an ML stack."
                            color: Theme.dim
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
                        onClicked: appController.archiveProject(appController.selectedProject.id)
                    }
                }
            }
        }
    }
}

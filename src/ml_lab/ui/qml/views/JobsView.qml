import QtQuick
import QtQuick.Controls
import QtQuick.Layouts
import "../components"

Item {
    ColumnLayout {
        anchors.fill: parent
        spacing: 16

        RowLayout {
            Layout.fillWidth: true

            ColumnLayout {
                spacing: 2

                Text {
                    text: "Jobs"
                    color: Theme.text
                    font.pixelSize: 26
                    font.weight: Font.Bold
                }

                Text {
                    text: "Long work runs outside the UI process and leaves structured evidence."
                    color: Theme.muted
                    font.pixelSize: 13
                }
            }

            Item {
                Layout.fillWidth: true
            }

            LabButton {
                text: "Run self-test worker"
                primary: true
                onClicked: appController.runSelfTest()
            }
        }

        Panel {
            Layout.fillWidth: true
            Layout.fillHeight: true

            ListView {
                id: list
                anchors.fill: parent
                anchors.margins: 10
                spacing: 7
                clip: true
                model: appController.jobs

                delegate: Rectangle {
                    required property var modelData

                    width: list.width
                    height: 96
                    radius: 9
                    color: Theme.surfaceAlt
                    border.color: Theme.border

                    RowLayout {
                        anchors.fill: parent
                        anchors.margins: 14
                        spacing: 12

                        ColumnLayout {
                            Layout.fillWidth: true
                            spacing: 5

                            RowLayout {
                                Text {
                                    text: modelData.taskType
                                    color: Theme.text2
                                    font.weight: Font.DemiBold
                                }

                                StatusPill {
                                    text: modelData.status
                                    tone: modelData.status === "COMPLETED" ? "good"
                                          : (modelData.status === "FAILED"
                                             || modelData.status === "INTERRUPTED") ? "bad" : "neutral"
                                }
                            }

                            ProgressBar {
                                from: 0
                                to: 1
                                value: modelData.progress
                                Layout.fillWidth: true
                            }

                            Text {
                                text: modelData.error || modelData.message
                                color: modelData.error ? Theme.badText : Theme.muted
                                elide: Text.ElideRight
                                Layout.fillWidth: true
                            }

                            Text {
                                visible: modelData.resultDigest.length > 0
                                text: "Result artifact  " + modelData.resultDigest.slice(0, 16) + "…"
                                color: Theme.dim
                                font.pixelSize: 10
                            }
                        }

                        LabButton {
                            text: "Cancel"
                            visible: modelData.status === "RUNNING" || modelData.status === "QUEUED"
                            onClicked: appController.cancelJob(modelData.id)
                        }
                    }
                }

                footer: Item {
                    width: list.width
                    height: appController.jobs.length === 0 ? 180 : 0

                    Text {
                        anchors.centerIn: parent
                        visible: appController.jobs.length === 0
                        text: "No jobs yet."
                        color: Theme.dim
                    }
                }
            }
        }
    }
}

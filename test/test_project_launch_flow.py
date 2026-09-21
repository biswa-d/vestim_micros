import importlib
import unittest


class ProjectLaunchFlowTests(unittest.TestCase):
    def test_project_startup_defaults_to_data_import_flow(self):
        config_manager = importlib.import_module("vestim.config_manager")

        startup = config_manager.get_project_startup_settings(None)

        self.assertEqual(startup["flow"], "training")
        self.assertTrue(startup["auto_open_data_import"])
        self.assertFalse(startup["auto_continue_to_augmentation"])

    def test_project_startup_respects_project_payload_flags(self):
        config_manager = importlib.import_module("vestim.config_manager")

        launch_context = {
            "project_payload": {
                "startup": {
                    "flow": "testing",
                    "auto_open_data_import": False,
                    "auto_continue_to_augmentation": True,
                }
            }
        }

        startup = config_manager.get_project_startup_settings(launch_context)

        self.assertEqual(startup["flow"], "testing")
        self.assertFalse(startup["auto_open_data_import"])
        self.assertTrue(startup["auto_continue_to_augmentation"])


if __name__ == "__main__":
    unittest.main()

# Copyright (c) 2023 - 2026 Chair for Design Automation, TUM
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

"""Shared browser player for schedule views.

The player owns the page, the playback controls, the time slider, the video
range, and video export. A compiler-specific drawing script supplies the data
preparation and one function that draws a given schedule time.
"""

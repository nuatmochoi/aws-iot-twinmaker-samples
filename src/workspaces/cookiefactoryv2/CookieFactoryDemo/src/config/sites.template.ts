// Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved. 2023
// SPDX-License-Identifier: Apache-2.0

/**
 * Project sites configuration.
 * RENAME THIS TEMPLATE TO `sites.ts`
 */

import type { SiteConfig, UserConfig } from '@/lib/types';

const sites: Record<string, SiteConfig[]> = {
  'user@cookiefactory': [
    {
      defaults: {
        panelIds: [],
        viewId: 'panel'
      },
      iottwinmaker: {
        sceneId: 'CookieFactory',
        workspaceId: '__FILL_IN__'
      },
      id: crypto.randomUUID(),
      location: '안성시 쿠키 제조 공장',
      name: '안성 제과 공장'
    }
  ]
};

export const SITE_TYPE = 'Facility';

export function getLookUpKey(user: UserConfig) {
  return user.email;
}

export default sites;

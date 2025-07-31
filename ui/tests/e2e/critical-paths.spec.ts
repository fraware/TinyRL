import { test, expect } from '@playwright/test';

test.describe('Critical User Paths', () => {
  test.beforeEach(async ({ page }) => {
    await page.goto('/');
  });

  test('should create a new project successfully', async ({ page }) => {
    // Navigate to projects page
    await page.click('text=Projects');
    
    // Click create project button
    await page.click('text=New Project');
    
    // Fill project form
    await page.fill('input[name="name"]', 'Test CartPole Project');
    await page.fill('textarea[name="description"]', 'A test project for CartPole environment');
    await page.selectOption('select[name="environment"]', 'CartPole-v1');
    await page.selectOption('select[name="algorithm"]', 'PPO');
    
    // Submit form
    await page.click('button[type="submit"]');
    
    // Verify project was created
    await expect(page.locator('text=Test CartPole Project')).toBeVisible();
    await expect(page.locator('text=CartPole-v1')).toBeVisible();
    await expect(page.locator('text=PPO')).toBeVisible();
  });

  test('should start a pipeline and monitor progress', async ({ page }) => {
    // Navigate to pipelines page
    await page.click('text=Pipelines');
    
    // Click new pipeline button
    await page.click('text=New Pipeline');
    
    // Select project
    await page.selectOption('select[name="project"]', 'CartPole PPO Agent');
    
    // Configure pipeline
    await page.fill('input[name="total_timesteps"]', '10000');
    await page.selectOption('select[name="algorithm"]', 'PPO');
    
    // Start pipeline
    await page.click('text=Start Pipeline');
    
    // Verify pipeline started
    await expect(page.locator('text=CartPole PPO Training')).toBeVisible();
    await expect(page.locator('text=running')).toBeVisible();
    
    // Wait for completion (simulated)
    await page.waitForSelector('text=completed', { timeout: 30000 });
    
    // Verify results
    await expect(page.locator('text=98.5%')).toBeVisible();
    await expect(page.locator('text=2048 bytes')).toBeVisible();
  });

  test('should flash device using wizard', async ({ page }) => {
    // Navigate to fleet page
    await page.click('text=Fleet');
    
    // Click add device button
    await page.click('text=Add Device');
    
    // Start flash wizard
    await page.click('text=Flash Device Wizard');
    
    // Step 1: Device detection
    await page.click('text=Detect Devices');
    await page.waitForSelector('text=Found 3 connected devices');
    
    // Select device
    await page.click('text=STM32 Nucleo-144');
    await page.click('text=Continue');
    
    // Step 2: Binary selection
    await page.click('text=CartPole PPO v1.0.0');
    await page.click('text=Continue');
    
    // Step 3: Power budget
    await page.fill('input[type="range"]', '50');
    await page.click('text=Continue');
    
    // Step 4: Flash device
    await page.click('text=Start Flash');
    
    // Verify flash progress
    await expect(page.locator('text=Flashing Progress')).toBeVisible();
    await expect(page.locator('text=100%')).toBeVisible({ timeout: 15000 });
    
    // Verify completion
    await expect(page.locator('text=Flash completed successfully')).toBeVisible();
  });

  test('should view and download artifacts', async ({ page }) => {
    // Navigate to artifacts page
    await page.click('text=Artifacts');
    
    // Verify artifacts are listed
    await expect(page.locator('text=cartpole-ppo-v1.0.0.bin')).toBeVisible();
    await expect(page.locator('text=98.5%')).toBeVisible();
    await expect(page.locator('text=verified')).toBeVisible();
    
    // Download artifact
    await page.click('button:has-text("Download")');
    
    // Verify download started
    const downloadPromise = page.waitForEvent('download');
    await downloadPromise;
  });

  test('should view proof verification details', async ({ page }) => {
    // Navigate to artifacts page
    await page.click('text=Artifacts');
    
    // Click view proof button
    await page.click('button:has-text("View Proof")');
    
    // Verify proof viewer opened
    await expect(page.locator('text=Proof Information')).toBeVisible();
    await expect(page.locator('text=Formal verification proof')).toBeVisible();
    
    // Verify proof content
    await expect(page.locator('text=Proof Content')).toBeVisible();
    await expect(page.locator('text=Copy Reference')).toBeVisible();
    await expect(page.locator('text=Download')).toBeVisible();
  });

  test('should monitor fleet devices', async ({ page }) => {
    // Navigate to fleet page
    await page.click('text=Fleet');
    
    // Verify device cards
    await expect(page.locator('text=STM32-Nucleo-144')).toBeVisible();
    await expect(page.locator('text=online')).toBeVisible();
    await expect(page.locator('text=98.5%')).toBeVisible();
    
    // Check device details
    await page.click('text=STM32-Nucleo-144');
    await expect(page.locator('text=Current Model')).toBeVisible();
    await expect(page.locator('text=CartPole PPO v1.0.0')).toBeVisible();
    
    // Verify performance metrics
    await expect(page.locator('text=Reward Trend')).toBeVisible();
    await expect(page.locator('text=5.2ms')).toBeVisible();
    await expect(page.locator('text=12.5mW')).toBeVisible();
  });

  test('should search and filter projects', async ({ page }) => {
    // Navigate to projects page
    await page.click('text=Projects');
    
    // Search for specific project
    await page.fill('input[placeholder="Search projects..."]', 'CartPole');
    
    // Verify filtered results
    await expect(page.locator('text=CartPole PPO Agent')).toBeVisible();
    await expect(page.locator('text=CartPole')).not.toBeVisible();
    
    // Clear search
    await page.fill('input[placeholder="Search projects..."]', '');
    
    // Verify all projects visible
    await expect(page.locator('text=CartPole PPO Agent')).toBeVisible();
    await expect(page.locator('text=LunarLander A2C')).toBeVisible();
  });

  test('should handle error states gracefully', async ({ page }) => {
    // Navigate to projects page
    await page.click('text=Projects');
    
    // Simulate network error
    await page.route('**/api/projects', route => route.abort());
    
    // Refresh page
    await page.reload();
    
    // Verify error state
    await expect(page.locator('text=No projects found')).toBeVisible();
    await expect(page.locator('text=Try adjusting your search terms')).toBeVisible();
  });

  test('should be keyboard accessible', async ({ page }) => {
    // Navigate to projects page
    await page.keyboard.press('Tab');
    await page.keyboard.press('Enter');
    
    // Navigate through projects using keyboard
    await page.keyboard.press('Tab');
    await page.keyboard.press('Tab');
    await page.keyboard.press('Enter');
    
    // Verify keyboard navigation works
    await expect(page.locator('text=Projects')).toBeVisible();
  });

  test('should be responsive on mobile', async ({ page }) => {
    // Set mobile viewport
    await page.setViewportSize({ width: 375, height: 667 });
    
    // Navigate to projects page
    await page.click('text=Projects');
    
    // Verify mobile layout
    await expect(page.locator('text=Projects')).toBeVisible();
    await expect(page.locator('button:has-text("New Project")')).toBeVisible();
    
    // Test mobile navigation
    await page.click('button[aria-label="Toggle sidebar"]');
    await expect(page.locator('text=Dashboard')).toBeVisible();
  });
}); 
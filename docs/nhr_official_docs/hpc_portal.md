# HPC Portal -- NHR@FAU

Source: https://doc.nhr.fau.de/hpc-portal/

## Login Options

The HPC-Portal employs single sign-on (SSO) for all user logins, supporting organizations participating in DFN-AAI or eduGAIN. Users select their organization from a dropdown and follow their identity provider's authentication process.

If an error message appears stating that "required attributes for correct account generation are missing," users should contact their local IT support, as organizational settings must be configured to enable proper identity transmission.

## The User Tab

This section displays accounts and project invitations. New accounts are created after a PI or technical contact sends an invitation and the recipient accepts it. **Account access becomes fully operational the following day.**

### SSH Key Requirements

Users must upload at least one public SSH key per account to access HPC systems:

1. Generate an SSH key pair locally
2. Select the desired account in the User tab
3. Click "Add new SSH key" under "Public SSH keys"
4. Accepted formats: RSA (4096+ bit), ECDSA (512+ bit), ED25519
5. Provide a meaningful alias and paste the public key content
6. Submit the key

Key distribution to all systems takes up to two hours.

The User tab also displays resource usage and provides a "Go to ClusterCockpit" button for detailed job monitoring.

**Note:** User data on HPC systems is deleted three months after account expiration.

## Profile Information

Users access profile settings by clicking their email address in the top right corner and selecting Profile. This area manages NHR newsletter subscriptions and displays export control and terms of use information.

## The Management Tab

Available only to PIs and technical contacts, this tab enables project management. Users can:

- View project details and associated accounts
- Create new accounts by sending invitations
- Edit account states (pending, approved, deleted, active, inactive) and validity periods
- Review compute resource usage per project or account

**Important:** The invitation email must match the IdP-transmitted email address or it won't be visible to recipients. Multiple invitations can be sent simultaneously using the "Invite multiple e-mail addresses" option.

## Roles Overview

### User

- Default role for SSO login
- Access: own profile, accounts, and monitoring data
- Permissions: SSH key uploads, ClusterCockpit (own jobs), JupyterHub

### Manager/PI

- Same permissions, limited to assigned projects
- Access: management tab, project invitations, accounts
- Permissions: user management, ClusterCockpit (all project jobs)

### Advisor

NHR@FAU support staff and Liaison Scientists

- Access: read-only project information
- Permissions: usage monitoring, ClusterCockpit (project jobs)

### Support

NHR@FAU support staff only

- Access: read-only all project data
- Permissions: usage monitoring, HPC-Portal troubleshooting, ClusterCockpit (all jobs/cluster state)

### Admin

NHR@FAU administrators only

- Full read/write access to all project data
- Permissions: project creation, role assignment, data export

# Job Monitoring with ClusterCockpit -- NHR@FAU

Source: https://doc.nhr.fau.de/job-monitoring-with-clustercockpit/

## Introduction

NHR@FAU provides ClusterCockpit, a web interface designed to monitor job performance. The tool grants HPC users access to hardware performance metrics including flop rates, main memory bandwidth, SIMD vectorization ratios, allocated memory capacity, as well as network and file IO metrics. Project managers can view jobs from team members working on the same project. ClusterCockpit is open-source and developed by NHR@FAU.

## Access

Access ClusterCockpit through an authenticated HPC Portal session. Within your account settings, locate and click the "Go to ClusterCockpit" button.

## Basic User Interface

The top navigation bar contains:
- Universal search functionality for job IDs, job names, and array job IDs
- Configuration options for layout and plot customization
- An information tooltip explaining search syntax

Job lists feature filtering and sorting capabilities. Users can customize which metrics display, with preferences saved per user and view.

## Available Views

### Landing Page
Displays all clusters with active user jobs, offering direct links to running and completed job lists for each cluster.

### My Jobs
Features a statistics overview showing aggregated job counts and duration/node usage histograms. The job list presents metadata and resource utilization plots with reference lines. Red or orange backgrounds indicate suboptimal resource usage.

Job-specific details appear after clicking a job ID. Note: Running jobs appear in listings a few minutes after starting.

### Tag View
Users can apply key/value tags to jobs for enriched metadata. This view allows filtering jobs by selected tags.

## Reporting Problems

Contact support at hpc-support@fau.de for ClusterCockpit service issues.

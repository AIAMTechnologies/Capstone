--
-- PostgreSQL database dump
--

\restrict sHe2Z43VLnnzBXEra80tLvvlQsqf6Ej6uVQM250TnIVJnu9UC3X3qLJS7D3zMU3

-- Dumped from database version 16.3
-- Dumped by pg_dump version 16.10 (Homebrew)

SET statement_timeout = 0;
SET lock_timeout = 0;
SET idle_in_transaction_session_timeout = 0;
SET client_encoding = 'UTF8';
SET standard_conforming_strings = on;
SELECT pg_catalog.set_config('search_path', '', false);
SET check_function_bodies = false;
SET xmloption = content;
SET client_min_messages = warning;
SET row_security = off;

--
-- Name: update_installer_capacity(); Type: FUNCTION; Schema: public; Owner: postgres
--

CREATE FUNCTION public.update_installer_capacity() RETURNS trigger
    LANGUAGE plpgsql
    AS $$
BEGIN
    -- When a lead is assigned
    IF NEW.assigned_installer_id IS NOT NULL AND 
       (OLD.assigned_installer_id IS NULL OR OLD.assigned_installer_id != NEW.assigned_installer_id) THEN
        
        -- Increment new installer's capacity
        UPDATE installers 
        SET current_capacity = current_capacity + 1 
        WHERE id = NEW.assigned_installer_id;
        
        -- Decrement old installer's capacity if there was one
        IF OLD.assigned_installer_id IS NOT NULL THEN
            UPDATE installers 
            SET current_capacity = GREATEST(current_capacity - 1, 0)
            WHERE id = OLD.assigned_installer_id;
        END IF;
        
        -- Set assigned_at timestamp
        NEW.assigned_at = CURRENT_TIMESTAMP;
    END IF;
    
    -- When a lead is completed or cancelled
    IF NEW.status IN ('completed', 'cancelled', 'lost') AND 
       OLD.status NOT IN ('completed', 'cancelled', 'lost') AND
       NEW.assigned_installer_id IS NOT NULL THEN
        
        -- Decrement installer's capacity
        UPDATE installers 
        SET current_capacity = GREATEST(current_capacity - 1, 0)
        WHERE id = NEW.assigned_installer_id;
        
        -- Increment total_jobs_completed for completed jobs
        IF NEW.status = 'completed' THEN
            UPDATE installers 
            SET total_jobs_completed = total_jobs_completed + 1
            WHERE id = NEW.assigned_installer_id;
            
            NEW.completed_at = CURRENT_TIMESTAMP;
        END IF;
    END IF;
    
    RETURN NEW;
END;
$$;


ALTER FUNCTION public.update_installer_capacity() OWNER TO postgres;

--
-- Name: update_updated_at_column(); Type: FUNCTION; Schema: public; Owner: postgres
--

CREATE FUNCTION public.update_updated_at_column() RETURNS trigger
    LANGUAGE plpgsql
    AS $$
BEGIN
    NEW.updated_at = CURRENT_TIMESTAMP;
    RETURN NEW;
END;
$$;


ALTER FUNCTION public.update_updated_at_column() OWNER TO postgres;

SET default_tablespace = '';

SET default_table_access_method = heap;

--
-- Name: installers; Type: TABLE; Schema: public; Owner: postgres
--

CREATE TABLE public.installers (
    id integer NOT NULL,
    name character varying(200) NOT NULL,
    city character varying(100),
    province character varying(50) NOT NULL,
    latitude numeric(10,6) NOT NULL,
    longitude numeric(10,6) NOT NULL,
    email character varying(100),
    phone character varying(20),
    service_radius_km integer DEFAULT 70,
    max_capacity integer DEFAULT 10,
    current_capacity integer DEFAULT 0,
    specialization character varying(50)[],
    is_active boolean DEFAULT true,
    success_rate numeric(3,2) DEFAULT 0.70,
    avg_completion_days integer DEFAULT 7,
    total_jobs_completed integer DEFAULT 0,
    rating numeric(2,1) DEFAULT 4.0,
    created_at timestamp without time zone DEFAULT CURRENT_TIMESTAMP,
    updated_at timestamp without time zone DEFAULT CURRENT_TIMESTAMP
);


ALTER TABLE public.installers OWNER TO postgres;

--
-- Name: leads; Type: TABLE; Schema: public; Owner: postgres
--

CREATE TABLE public.leads (
    id integer NOT NULL,
    name character varying(100) NOT NULL,
    email character varying(100) NOT NULL,
    phone character varying(20),
    address character varying(255) NOT NULL,
    city character varying(100) NOT NULL,
    province character varying(50) NOT NULL,
    postal_code character varying(10),
    latitude numeric(10,6),
    longitude numeric(10,6),
    job_type character varying(20) NOT NULL,
    comments text,
    assigned_installer_id integer,
    allocation_score numeric(8,2),
    distance_to_installer_km numeric(8,2),
    estimated_travel_time_minutes integer,
    status character varying(20) DEFAULT 'active'::character varying,
    created_at timestamp without time zone DEFAULT CURRENT_TIMESTAMP,
    assigned_at timestamp without time zone,
    updated_at timestamp without time zone DEFAULT CURRENT_TIMESTAMP,
    completed_at timestamp without time zone,
    source character varying(50) DEFAULT 'website'::character varying,
    priority character varying(10) DEFAULT 'normal'::character varying,
    CONSTRAINT leads_job_type_check CHECK (((job_type)::text = ANY ((ARRAY['residential'::character varying, 'commercial'::character varying])::text[]))),
    CONSTRAINT leads_priority_check CHECK (((priority)::text = ANY ((ARRAY['low'::character varying, 'normal'::character varying, 'high'::character varying, 'urgent'::character varying])::text[]))),
    CONSTRAINT leads_status_check CHECK (((status)::text = ANY ((ARRAY['active'::character varying, 'converted'::character varying, 'dead'::character varying])::text[])))
);


ALTER TABLE public.leads OWNER TO postgres;

--
-- Name: active_leads_detail; Type: VIEW; Schema: public; Owner: postgres
--

CREATE VIEW public.active_leads_detail AS
 SELECT l.id,
    l.name AS client_name,
    l.email,
    l.phone,
    l.city AS client_city,
    l.province AS client_province,
    l.job_type,
    l.status,
    l.allocation_score,
    l.distance_to_installer_km,
    l.estimated_travel_time_minutes,
    l.created_at,
    i.name AS installer_name,
    i.city AS installer_city,
    i.email AS installer_email,
    i.phone AS installer_phone
   FROM (public.leads l
     LEFT JOIN public.installers i ON ((l.assigned_installer_id = i.id)))
  WHERE ((l.status)::text = ANY ((ARRAY['active'::character varying, 'dead'::character varying, 'converted'::character varying])::text[]))
  ORDER BY l.created_at DESC;


ALTER VIEW public.active_leads_detail OWNER TO postgres;

--
-- Name: admin_users; Type: TABLE; Schema: public; Owner: postgres
--

CREATE TABLE public.admin_users (
    id integer NOT NULL,
    username character varying(50) NOT NULL,
    password_hash character varying(255) NOT NULL,
    email character varying(100) NOT NULL,
    last_name character varying(100),
    role character varying(20) DEFAULT 'admin'::character varying,
    is_active boolean DEFAULT true,
    created_at timestamp without time zone DEFAULT CURRENT_TIMESTAMP,
    last_login timestamp without time zone
);


ALTER TABLE public.admin_users OWNER TO postgres;

--
-- Name: admin_users_id_seq; Type: SEQUENCE; Schema: public; Owner: postgres
--

CREATE SEQUENCE public.admin_users_id_seq
    AS integer
    START WITH 1
    INCREMENT BY 1
    NO MINVALUE
    NO MAXVALUE
    CACHE 1;


ALTER SEQUENCE public.admin_users_id_seq OWNER TO postgres;

--
-- Name: admin_users_id_seq; Type: SEQUENCE OWNED BY; Schema: public; Owner: postgres
--

ALTER SEQUENCE public.admin_users_id_seq OWNED BY public.admin_users.id;


--
-- Name: installers_id_seq; Type: SEQUENCE; Schema: public; Owner: postgres
--

CREATE SEQUENCE public.installers_id_seq
    AS integer
    START WITH 1
    INCREMENT BY 1
    NO MINVALUE
    NO MAXVALUE
    CACHE 1;


ALTER SEQUENCE public.installers_id_seq OWNER TO postgres;

--
-- Name: installers_id_seq; Type: SEQUENCE OWNED BY; Schema: public; Owner: postgres
--

ALTER SEQUENCE public.installers_id_seq OWNED BY public.installers.id;


--
-- Name: lead_outcomes; Type: TABLE; Schema: public; Owner: postgres
--

CREATE TABLE public.lead_outcomes (
    id integer NOT NULL,
    submit_date timestamp without time zone,
    first_name character varying(100),
    last_name character varying(100),
    company_name character varying(255),
    address1 character varying(255),
    city character varying(100),
    province character varying(10),
    postal character varying(20),
    dealer_name character varying(255),
    project_type character varying(50),
    product_type character varying(100),
    square_footage character varying(50),
    current_status character varying(50),
    job_won_date timestamp without time zone,
    value_of_order numeric(10,2),
    job_lost_date timestamp without time zone,
    reason character varying(255),
    created_at timestamp without time zone DEFAULT CURRENT_TIMESTAMP,
    updated_at timestamp without time zone DEFAULT CURRENT_TIMESTAMP
);


ALTER TABLE public.lead_outcomes OWNER TO postgres;

--
-- Name: TABLE lead_outcomes; Type: COMMENT; Schema: public; Owner: postgres
--

COMMENT ON TABLE public.lead_outcomes IS 'Lead outcomes and conversion tracking from 3M data';


--
-- Name: COLUMN lead_outcomes.submit_date; Type: COMMENT; Schema: public; Owner: postgres
--

COMMENT ON COLUMN public.lead_outcomes.submit_date IS 'Date when lead was submitted';


--
-- Name: COLUMN lead_outcomes.square_footage; Type: COMMENT; Schema: public; Owner: postgres
--

COMMENT ON COLUMN public.lead_outcomes.square_footage IS 'Project size range';


--
-- Name: COLUMN lead_outcomes.current_status; Type: COMMENT; Schema: public; Owner: postgres
--

COMMENT ON COLUMN public.lead_outcomes.current_status IS 'Status: New, Dead Lead, Converted Sale, Called, Follow Up, Client Reviewing/Undecided';


--
-- Name: COLUMN lead_outcomes.value_of_order; Type: COMMENT; Schema: public; Owner: postgres
--

COMMENT ON COLUMN public.lead_outcomes.value_of_order IS 'Revenue from converted sales';


--
-- Name: COLUMN lead_outcomes.reason; Type: COMMENT; Schema: public; Owner: postgres
--

COMMENT ON COLUMN public.lead_outcomes.reason IS 'Reason for dead leads: Competition, Customer Never Returned Call, Product Not Needed, Other';


--
-- Name: lead_outcomes_backup; Type: TABLE; Schema: public; Owner: postgres
--

CREATE TABLE public.lead_outcomes_backup (
    id integer,
    lead_id integer,
    installer_id integer,
    outcome character varying(20),
    outcome_date timestamp without time zone,
    response_time_hours integer,
    quote_value numeric(10,2),
    actual_revenue numeric(10,2),
    completion_time_days integer,
    customer_satisfaction integer,
    would_recommend boolean,
    notes text,
    dead_reason character varying(100),
    created_at timestamp without time zone
);


ALTER TABLE public.lead_outcomes_backup OWNER TO postgres;

--
-- Name: lead_outcomes_id_seq; Type: SEQUENCE; Schema: public; Owner: postgres
--

CREATE SEQUENCE public.lead_outcomes_id_seq
    AS integer
    START WITH 1
    INCREMENT BY 1
    NO MINVALUE
    NO MAXVALUE
    CACHE 1;


ALTER SEQUENCE public.lead_outcomes_id_seq OWNER TO postgres;

--
-- Name: lead_outcomes_id_seq; Type: SEQUENCE OWNED BY; Schema: public; Owner: postgres
--

ALTER SEQUENCE public.lead_outcomes_id_seq OWNED BY public.lead_outcomes.id;


--
-- Name: lead_outcomes_summary; Type: VIEW; Schema: public; Owner: postgres
--

CREATE VIEW public.lead_outcomes_summary AS
 SELECT count(*) AS total_leads,
    count(
        CASE
            WHEN ((current_status)::text = 'Converted Sale'::text) THEN 1
            ELSE NULL::integer
        END) AS converted_leads,
    count(
        CASE
            WHEN ((current_status)::text = 'Dead Lead'::text) THEN 1
            ELSE NULL::integer
        END) AS dead_leads,
    round((((count(
        CASE
            WHEN ((current_status)::text = 'Converted Sale'::text) THEN 1
            ELSE NULL::integer
        END))::numeric / (NULLIF(count(*), 0))::numeric) * (100)::numeric), 2) AS conversion_rate_percent,
    sum(value_of_order) AS total_revenue,
    avg(value_of_order) AS avg_order_value
   FROM public.lead_outcomes;


ALTER VIEW public.lead_outcomes_summary OWNER TO postgres;

--
-- Name: leads_id_seq; Type: SEQUENCE; Schema: public; Owner: postgres
--

CREATE SEQUENCE public.leads_id_seq
    AS integer
    START WITH 1
    INCREMENT BY 1
    NO MINVALUE
    NO MAXVALUE
    CACHE 1;


ALTER SEQUENCE public.leads_id_seq OWNER TO postgres;

--
-- Name: leads_id_seq; Type: SEQUENCE OWNED BY; Schema: public; Owner: postgres
--

ALTER SEQUENCE public.leads_id_seq OWNED BY public.leads.id;


--
-- Name: admin_users id; Type: DEFAULT; Schema: public; Owner: postgres
--

ALTER TABLE ONLY public.admin_users ALTER COLUMN id SET DEFAULT nextval('public.admin_users_id_seq'::regclass);


--
-- Name: installers id; Type: DEFAULT; Schema: public; Owner: postgres
--

ALTER TABLE ONLY public.installers ALTER COLUMN id SET DEFAULT nextval('public.installers_id_seq'::regclass);


--
-- Name: lead_outcomes id; Type: DEFAULT; Schema: public; Owner: postgres
--

ALTER TABLE ONLY public.lead_outcomes ALTER COLUMN id SET DEFAULT nextval('public.lead_outcomes_id_seq'::regclass);


--
-- Name: leads id; Type: DEFAULT; Schema: public; Owner: postgres
--

ALTER TABLE ONLY public.leads ALTER COLUMN id SET DEFAULT nextval('public.leads_id_seq'::regclass);


--
-- Data for Name: admin_users; Type: TABLE DATA; Schema: public; Owner: postgres
--

COPY public.admin_users (id, username, password_hash, email, last_name, role, is_active, created_at, last_login) FROM stdin;
4	admin2	$2y$10$0udE6FS.fDgoTRvit8ytC.cxe/tJwdLhuLh9CbXrVdlnsmq2plxtK	admin2@company.com	Secondary Administrator	admin	t	2025-10-30 20:45:20.245091	\N
3	admin1	$2y$10$0udE6FS.fDgoTRvit8ytC.cxe/tJwdLhuLh9CbXrVdlnsmq2plxtK	admin1@company.com	Primary Administrator	admin	t	2025-10-30 20:45:20.245091	2025-11-06 01:23:55.600276
\.


--
-- Data for Name: installers; Type: TABLE DATA; Schema: public; Owner: postgres
--

COPY public.installers (id, name, city, province, latitude, longitude, email, phone, service_radius_km, max_capacity, current_capacity, specialization, is_active, success_rate, avg_completion_days, total_jobs_completed, rating, created_at, updated_at) FROM stdin;
1	Titan Window Films Ltd.	Victoria	BC	48.435299	-123.491242	\N	\N	70	10	0	{residential,commercial}	t	0.70	7	0	4.0	2025-10-30 17:53:36.046302	2025-10-30 17:53:36.046302
2	Titan Window Films Ltd.	Vancouver	BC	49.273114	-123.100348	\N	\N	70	10	0	{residential,commercial}	t	0.70	7	0	4.0	2025-10-30 17:53:36.046302	2025-10-30 17:53:36.046302
3	TINT'D Film & Graphic Solutions	Langley	BC	49.112540	-122.653360	\N	\N	70	10	0	{residential,commercial}	t	0.70	7	0	4.0	2025-10-30 17:53:36.046302	2025-10-30 17:53:36.046302
4	LR Window Films	North Vancouver	BC	49.304840	-122.958268	\N	\N	70	10	0	{residential,commercial}	t	0.70	7	0	4.0	2025-10-30 17:53:36.046302	2025-10-30 17:53:36.046302
5	Urban Window Films	Kelowna	BC	49.880000	-119.440000	\N	\N	70	10	0	{residential,commercial}	t	0.70	7	0	4.0	2025-10-30 17:53:36.046302	2025-10-30 17:53:36.046302
6	Signwriter	Cranbrook	BC	49.523389	-115.761820	\N	\N	70	10	0	{residential,commercial}	t	0.70	7	0	4.0	2025-10-30 17:53:36.046302	2025-10-30 17:53:36.046302
7	SignTek Industries	Prince George	BC	53.913970	-122.735200	\N	\N	70	10	0	{commercial}	t	0.70	7	0	4.0	2025-10-30 17:53:36.046302	2025-10-30 17:53:36.046302
8	Carbon Graphics Group	Edmonton	AB	53.543300	-113.500700	\N	\N	70	10	0	{residential,commercial}	t	0.70	7	0	4.0	2025-10-30 17:53:36.046302	2025-10-30 17:53:36.046302
9	Royal Glass	Sylvan Lake	AB	52.311000	-114.100000	\N	\N	70	10	0	{residential,commercial}	t	0.70	7	0	4.0	2025-10-30 17:53:36.046302	2025-10-30 17:53:36.046302
10	South Country Glass Ltd.	Medicine Hat	AB	50.039000	-110.674000	\N	\N	70	10	0	{residential,commercial}	t	0.70	7	0	4.0	2025-10-30 17:53:36.046302	2025-10-30 17:53:36.046302
11	ATG Architectural Tint & Graphics	Saskatoon	SK	52.174000	-106.647000	\N	\N	70	10	0	{residential,commercial}	t	0.70	7	0	4.0	2025-10-30 17:53:36.046302	2025-10-30 17:53:36.046302
12	Artek Film Solutions	Saskatoon	SK	52.130000	-106.660000	\N	\N	70	10	0	{residential,commercial}	t	0.70	7	0	4.0	2025-10-30 17:53:36.046302	2025-10-30 17:53:36.046302
13	D&D Sign & Graphic	Regina	SK	50.433000	-104.500000	\N	\N	70	10	0	{residential,commercial}	t	0.70	7	0	4.0	2025-10-30 17:53:36.046302	2025-10-30 17:53:36.046302
14	Artek Film Solutions	Regina	SK	50.454000	-104.618000	\N	\N	70	10	0	{residential,commercial}	t	0.70	7	0	4.0	2025-10-30 17:53:36.046302	2025-10-30 17:53:36.046302
15	VBG Distributors Ltd.	Winnipeg	MB	49.884000	-97.058000	\N	\N	70	10	0	{residential,commercial}	t	0.70	7	0	4.0	2025-10-30 17:53:36.046302	2025-10-30 17:53:36.046302
16	Total Tint	Toronto	ON	43.602000	-79.545000	\N	\N	70	10	0	{residential,commercial}	t	0.70	7	0	4.0	2025-10-30 17:53:36.046302	2025-10-30 17:53:36.046302
17	Evolution Window Films	Grimsby	ON	43.195000	-79.557000	\N	\N	70	10	0	{residential,commercial}	t	0.70	7	0	4.0	2025-10-30 17:53:36.046302	2025-10-30 17:53:36.046302
18	Peak Window Films	Burlington	ON	43.308000	-79.855000	\N	\N	70	10	0	{residential,commercial}	t	0.70	7	0	4.0	2025-10-30 17:53:36.046302	2025-10-30 17:53:36.046302
19	Lindian Enterprises Ltd.	Ottawa	ON	45.334000	-75.805000	\N	\N	70	10	0	{residential,commercial}	t	0.70	7	0	4.0	2025-10-30 17:53:36.046302	2025-10-30 17:53:36.046302
20	Capital Solar and Security	Ottawa	ON	45.423000	-75.610000	\N	\N	70	10	0	{residential,commercial}	t	0.70	7	0	4.0	2025-10-30 17:53:36.046302	2025-10-30 17:53:36.046302
21	Franklin Tint	Oshawa	ON	43.885000	-78.856000	\N	\N	70	10	0	{residential,commercial}	t	0.70	7	0	4.0	2025-10-30 17:53:36.046302	2025-10-30 17:53:36.046302
22	TriCounty Window Film Solutions	Woodstock	ON	43.152000	-80.754000	\N	\N	70	10	0	{residential,commercial}	t	0.70	7	0	4.0	2025-10-30 17:53:36.046302	2025-10-30 17:53:36.046302
23	TriCounty Window Film Solutions	Cambridge	ON	43.369000	-80.312000	\N	\N	70	10	0	{residential,commercial}	t	0.70	7	0	4.0	2025-10-30 17:53:36.046302	2025-10-30 17:53:36.046302
24	Peak Window Films	Kitchener	ON	43.434000	-80.472000	\N	\N	70	10	0	{residential,commercial}	t	0.70	7	0	4.0	2025-10-30 17:53:36.046302	2025-10-30 17:53:36.046302
25	Blissful Blinds Inc.	Goderich	ON	43.742000	-81.707000	\N	\N	70	10	0	{residential}	t	0.70	7	0	4.0	2025-10-30 17:53:36.046302	2025-10-30 17:53:36.046302
26	Unique Window Films	Barrie	ON	44.428000	-79.664000	\N	\N	70	10	0	{residential,commercial}	t	0.70	7	0	4.0	2025-10-30 17:53:36.046302	2025-10-30 17:53:36.046302
27	Crystal's Glass Tinting	Barrie	ON	44.389000	-79.708000	\N	\N	70	10	0	{residential,commercial}	t	0.70	7	0	4.0	2025-10-30 17:53:36.046302	2025-10-30 17:53:36.046302
28	Glass Canada Ltd.	London	ON	42.935000	-81.248000	\N	\N	70	10	0	{residential,commercial}	t	0.70	7	0	4.0	2025-10-30 17:53:36.046302	2025-10-30 17:53:36.046302
29	Windsor Window Imaging Inc.	Windsor	ON	42.276000	-83.061000	\N	\N	70	10	0	{residential,commercial}	t	0.70	7	0	4.0	2025-10-30 17:53:36.046302	2025-10-30 17:53:36.046302
30	Blue Coast Architectural Finishes	Sarnia	ON	42.974000	-82.406000	\N	\N	70	10	0	{commercial}	t	0.70	7	0	4.0	2025-10-30 17:53:36.046302	2025-10-30 17:53:36.046302
31	Verticals N' Visions	Thunder Bay	ON	48.389000	-89.244000	\N	\N	70	10	0	{residential,commercial}	t	0.70	7	0	4.0	2025-10-30 17:53:36.046302	2025-10-30 17:53:36.046302
32	Price Window Films	North Bay	ON	46.311000	-79.468000	\N	\N	70	10	0	{residential,commercial}	t	0.70	7	0	4.0	2025-10-30 17:53:36.046302	2025-10-30 17:53:36.046302
33	Sudbury Window Tinting	Sudbury	ON	46.540000	-80.882000	\N	\N	70	10	0	{residential,commercial}	t	0.70	7	0	4.0	2025-10-30 17:53:36.046302	2025-10-30 17:53:36.046302
34	Jet Signs	Kingston	ON	44.248000	-76.571000	\N	\N	70	10	0	{commercial}	t	0.70	7	0	4.0	2025-10-30 17:53:36.046302	2025-10-30 17:53:36.046302
35	Smart Grafix	Timmins	ON	48.460000	-81.339000	\N	\N	70	10	0	{commercial}	t	0.70	7	0	4.0	2025-10-30 17:53:36.046302	2025-10-30 17:53:36.046302
36	Glass Employees Ltd.	Sault Ste. Marie	ON	46.526000	-84.300000	\N	\N	70	10	0	{residential,commercial}	t	0.70	7	0	4.0	2025-10-30 17:53:36.046302	2025-10-30 17:53:36.046302
37	Shade Window Films Inc.	Augusta	ON	44.738000	-75.546000	\N	\N	70	10	0	{residential,commercial}	t	0.70	7	0	4.0	2025-10-30 17:53:36.046302	2025-10-30 17:53:36.046302
38	Berkayly	Montréal	QC	45.588000	-73.612000	\N	\N	70	10	0	{residential,commercial}	t	0.70	7	0	4.0	2025-10-30 17:53:36.046302	2025-10-30 17:53:36.046302
39	Techteinte Bâtiment	Laval	QC	45.600000	-73.791000	\N	\N	70	10	0	{residential,commercial}	t	0.70	7	0	4.0	2025-10-30 17:53:36.046302	2025-10-30 17:53:36.046302
40	Stiick Pellicule sur fenêtre	Repentigny	QC	45.744000	-73.443000	\N	\N	70	10	0	{residential,commercial}	t	0.70	7	0	4.0	2025-10-30 17:53:36.046302	2025-10-30 17:53:36.046302
41	Berkayly	Shefford	QC	45.369000	-72.538000	\N	\N	70	10	0	{residential,commercial}	t	0.70	7	0	4.0	2025-10-30 17:53:36.046302	2025-10-30 17:53:36.046302
42	Protech-Sol	Charlesbourg	QC	46.876000	-71.274000	\N	\N	70	10	0	{residential,commercial}	t	0.70	7	0	4.0	2025-10-30 17:53:36.046302	2025-10-30 17:53:36.046302
43	Lindian Enterprises Ltd.	Gatineau	QC	45.484000	-75.641000	\N	\N	70	10	0	{residential,commercial}	t	0.70	7	0	4.0	2025-10-30 17:53:36.046302	2025-10-30 17:53:36.046302
44	Capital Solar and Security	Gatineau	QC	45.429000	-75.803000	\N	\N	70	10	0	{residential,commercial}	t	0.70	7	0	4.0	2025-10-30 17:53:36.046302	2025-10-30 17:53:36.046302
45	Lettrage Express	Chicoutimi-Nord	QC	48.460000	-71.065000	\N	\N	70	10	0	{commercial}	t	0.70	7	0	4.0	2025-10-30 17:53:36.046302	2025-10-30 17:53:36.046302
46	Vitrerie KRT	Rivière-du-Loup	QC	47.843000	-69.533000	\N	\N	70	10	0	{residential,commercial}	t	0.70	7	0	4.0	2025-10-30 17:53:36.046302	2025-10-30 17:53:36.046302
47	Maritime Window Film Specialists	Moncton	NB	46.087000	-64.811000	\N	\N	70	10	0	{residential,commercial}	t	0.70	7	0	4.0	2025-10-30 17:53:36.046302	2025-10-30 17:53:36.046302
48	Leonard Film and Graphics	Saint John	NB	45.292000	-66.037000	\N	\N	70	10	0	{residential,commercial}	t	0.70	7	0	4.0	2025-10-30 17:53:36.046302	2025-10-30 17:53:36.046302
49	Just Add Color Inc.	Halifax	NS	44.706000	-63.661000	\N	\N	70	10	0	{residential,commercial}	t	0.70	7	0	4.0	2025-10-30 17:53:36.046302	2025-10-30 17:53:36.046302
50	Tucker Window Films	St. John's	NL	47.570000	-52.722000	\N	\N	70	10	0	{residential,commercial}	t	0.70	7	0	4.0	2025-10-30 17:53:36.046302	2025-10-30 17:53:36.046302
\.


--
-- Data for Name: lead_outcomes; Type: TABLE DATA; Schema: public; Owner: postgres
--

COPY public.lead_outcomes (id, submit_date, first_name, last_name, company_name, address1, city, province, postal, dealer_name, project_type, product_type, square_footage, current_status, job_won_date, value_of_order, job_lost_date, reason, created_at, updated_at) FROM stdin;
\.


--
-- Data for Name: lead_outcomes_backup; Type: TABLE DATA; Schema: public; Owner: postgres
--

COPY public.lead_outcomes_backup (id, lead_id, installer_id, outcome, outcome_date, response_time_hours, quote_value, actual_revenue, completion_time_days, customer_satisfaction, would_recommend, notes, dead_reason, created_at) FROM stdin;
\.


--
-- Data for Name: leads; Type: TABLE DATA; Schema: public; Owner: postgres
--

COPY public.leads (id, name, email, phone, address, city, province, postal_code, latitude, longitude, job_type, comments, assigned_installer_id, allocation_score, distance_to_installer_km, estimated_travel_time_minutes, status, created_at, assigned_at, updated_at, completed_at, source, priority) FROM stdin;
1	John Smith	john.smith@email.com	416-555-0123	123 Main St	Toronto	ON	\N	43.653200	-79.383200	residential	Need window tinting for home office	\N	\N	\N	\N	active	2025-10-30 18:14:49.115408	\N	2025-10-30 18:14:49.115408	\N	website	normal
3	Jane Doe	jane.doe@email.com	514-555-0789	789 Rue Principale	Montréal	QC	\N	45.501700	-73.567300	residential	Privacy film for bathroom windows	\N	\N	\N	\N	active	2025-10-30 18:14:49.115408	\N	2025-10-30 18:14:49.115408	\N	website	normal
17	Test User	test@example.com	416-555-1234	100 King Street West	Toronto	ON	M5X 1A1	43.648768	-79.381692	residential	Testing if lead appears in list	16	1.85	14.14	14	converted	2025-10-30 21:26:41.278158	2025-10-30 21:26:41.278158	2025-10-30 21:35:25.204183	\N	website	normal
18	Pizza pizza	marie.t@example.com	514-555-9012	10 Thorndale Crescent	Hamilton	ON	L8S 3K2	43.260307	-79.924171	commercial	Storefront window film	18	-0.45	7.71	8	converted	2025-10-30 21:30:37.483141	2025-10-30 21:30:37.483141	2025-10-31 00:52:47.878171	\N	website	normal
19	Gino pizza	marie.t@example.com	514-555-9012	58 Bayshore Drive	Ottawa	ON	K2B 6M9	45.349825	-75.806142	commercial	Storefront window film	19	-2.57	1.76	2	active	2025-10-31 01:14:37.089843	2025-10-31 01:14:37.089843	2025-10-31 01:14:37.089843	\N	website	normal
20	Gino pizza	marie.t@example.com	514-555-9012	58 Bayshore Drive	Ottawa	ON	K2B 6M9	45.349825	-75.806142	commercial	Storefront window film	19	-2.57	1.76	2	active	2025-10-31 01:53:57.697996	2025-10-31 01:53:57.697996	2025-10-31 01:53:57.697996	\N	website	normal
21	Fit4less	scott@bayshore.ca	3653667081	58 Bayshore Drive	Ottawa	ON	K2B 6M9	45.349825	-75.806142	commercial		19	-2.57	1.76	2	active	2025-10-31 01:59:06.925813	2025-10-31 01:59:06.925813	2025-10-31 01:59:06.925813	\N	website	normal
22	Food basics	nl@f4d.ca	3653667081	230 Signal Hill Road	St. John's	NL	A1C 5M9	47.570344	-52.687224	residential		50	-2.27	2.61	3	active	2025-10-31 02:55:11.897757	2025-10-31 02:55:11.897757	2025-10-31 02:55:11.897757	\N	website	normal
2	ABC Corporation	contact@abccorp.com	604-555-0456	456 Business Ave	Vancouver	BC	\N	49.282700	-123.120700	commercial	Full office building window treatment	\N	\N	\N	\N	dead	2025-10-30 18:14:49.115408	\N	2025-10-31 02:55:34.528449	\N	website	normal
23	Fit4less	nl@f4d.ca	3653667081	10 Thorndale Crescent	Hamilton	ON	L8S 3K2	43.260307	-79.924171	residential		18	-0.45	7.71	8	active	2025-11-04 17:48:33.659895	2025-11-04 17:48:33.659895	2025-11-04 17:48:33.659895	\N	website	normal
24	Fit4less	nl@f4d.ca	3653667081	10 3 Street	Kleefeld	MB	R0A 0V1	49.501327	-96.877850	residential		15	12.69	44.48	44	converted	2025-11-04 18:19:44.722596	2025-11-04 18:19:44.722596	2025-11-04 18:30:03.118567	\N	website	normal
\.


--
-- Name: admin_users_id_seq; Type: SEQUENCE SET; Schema: public; Owner: postgres
--

SELECT pg_catalog.setval('public.admin_users_id_seq', 4, true);


--
-- Name: installers_id_seq; Type: SEQUENCE SET; Schema: public; Owner: postgres
--

SELECT pg_catalog.setval('public.installers_id_seq', 50, true);


--
-- Name: lead_outcomes_id_seq; Type: SEQUENCE SET; Schema: public; Owner: postgres
--

SELECT pg_catalog.setval('public.lead_outcomes_id_seq', 1, false);


--
-- Name: leads_id_seq; Type: SEQUENCE SET; Schema: public; Owner: postgres
--

SELECT pg_catalog.setval('public.leads_id_seq', 24, true);


--
-- Name: admin_users admin_users_email_key; Type: CONSTRAINT; Schema: public; Owner: postgres
--

ALTER TABLE ONLY public.admin_users
    ADD CONSTRAINT admin_users_email_key UNIQUE (email);


--
-- Name: admin_users admin_users_pkey; Type: CONSTRAINT; Schema: public; Owner: postgres
--

ALTER TABLE ONLY public.admin_users
    ADD CONSTRAINT admin_users_pkey PRIMARY KEY (id);


--
-- Name: admin_users admin_users_username_key; Type: CONSTRAINT; Schema: public; Owner: postgres
--

ALTER TABLE ONLY public.admin_users
    ADD CONSTRAINT admin_users_username_key UNIQUE (username);


--
-- Name: installers installers_pkey; Type: CONSTRAINT; Schema: public; Owner: postgres
--

ALTER TABLE ONLY public.installers
    ADD CONSTRAINT installers_pkey PRIMARY KEY (id);


--
-- Name: lead_outcomes lead_outcomes_pkey; Type: CONSTRAINT; Schema: public; Owner: postgres
--

ALTER TABLE ONLY public.lead_outcomes
    ADD CONSTRAINT lead_outcomes_pkey PRIMARY KEY (id);


--
-- Name: leads leads_pkey; Type: CONSTRAINT; Schema: public; Owner: postgres
--

ALTER TABLE ONLY public.leads
    ADD CONSTRAINT leads_pkey PRIMARY KEY (id);


--
-- Name: idx_installers_active; Type: INDEX; Schema: public; Owner: postgres
--

CREATE INDEX idx_installers_active ON public.installers USING btree (is_active);


--
-- Name: idx_installers_location; Type: INDEX; Schema: public; Owner: postgres
--

CREATE INDEX idx_installers_location ON public.installers USING btree (latitude, longitude);


--
-- Name: idx_installers_province; Type: INDEX; Schema: public; Owner: postgres
--

CREATE INDEX idx_installers_province ON public.installers USING btree (province);


--
-- Name: idx_lead_outcomes_current_status; Type: INDEX; Schema: public; Owner: postgres
--

CREATE INDEX idx_lead_outcomes_current_status ON public.lead_outcomes USING btree (current_status);


--
-- Name: idx_lead_outcomes_dealer_name; Type: INDEX; Schema: public; Owner: postgres
--

CREATE INDEX idx_lead_outcomes_dealer_name ON public.lead_outcomes USING btree (dealer_name);


--
-- Name: idx_lead_outcomes_project_type; Type: INDEX; Schema: public; Owner: postgres
--

CREATE INDEX idx_lead_outcomes_project_type ON public.lead_outcomes USING btree (project_type);


--
-- Name: idx_lead_outcomes_province; Type: INDEX; Schema: public; Owner: postgres
--

CREATE INDEX idx_lead_outcomes_province ON public.lead_outcomes USING btree (province);


--
-- Name: idx_lead_outcomes_submit_date; Type: INDEX; Schema: public; Owner: postgres
--

CREATE INDEX idx_lead_outcomes_submit_date ON public.lead_outcomes USING btree (submit_date);


--
-- Name: idx_leads_created_at; Type: INDEX; Schema: public; Owner: postgres
--

CREATE INDEX idx_leads_created_at ON public.leads USING btree (created_at DESC);


--
-- Name: idx_leads_installer; Type: INDEX; Schema: public; Owner: postgres
--

CREATE INDEX idx_leads_installer ON public.leads USING btree (assigned_installer_id);


--
-- Name: idx_leads_job_type; Type: INDEX; Schema: public; Owner: postgres
--

CREATE INDEX idx_leads_job_type ON public.leads USING btree (job_type);


--
-- Name: idx_leads_location; Type: INDEX; Schema: public; Owner: postgres
--

CREATE INDEX idx_leads_location ON public.leads USING btree (latitude, longitude);


--
-- Name: idx_leads_province; Type: INDEX; Schema: public; Owner: postgres
--

CREATE INDEX idx_leads_province ON public.leads USING btree (province);


--
-- Name: idx_leads_status; Type: INDEX; Schema: public; Owner: postgres
--

CREATE INDEX idx_leads_status ON public.leads USING btree (status);


--
-- Name: leads update_installer_capacity_trigger; Type: TRIGGER; Schema: public; Owner: postgres
--

CREATE TRIGGER update_installer_capacity_trigger BEFORE UPDATE ON public.leads FOR EACH ROW EXECUTE FUNCTION public.update_installer_capacity();


--
-- Name: leads update_leads_updated_at; Type: TRIGGER; Schema: public; Owner: postgres
--

CREATE TRIGGER update_leads_updated_at BEFORE UPDATE ON public.leads FOR EACH ROW EXECUTE FUNCTION public.update_updated_at_column();


--
-- Name: leads leads_assigned_installer_id_fkey; Type: FK CONSTRAINT; Schema: public; Owner: postgres
--

ALTER TABLE ONLY public.leads
    ADD CONSTRAINT leads_assigned_installer_id_fkey FOREIGN KEY (assigned_installer_id) REFERENCES public.installers(id);


--
-- PostgreSQL database dump complete
--

\unrestrict sHe2Z43VLnnzBXEra80tLvvlQsqf6Ej6uVQM250TnIVJnu9UC3X3qLJS7D3zMU3


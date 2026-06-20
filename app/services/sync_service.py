# app/services/sync_service.py
from app.core.config import settings
from app.services.paper_service import PaperService
from app.services.vector_store import VectorStoreService
from supabase import AsyncClient

class SyncService:
    def __init__(
        self,
        paper_service: PaperService,
        vector_store: VectorStoreService,
        supabase: AsyncClient,
    ):
        self.paper_service = paper_service
        self.vector_store = vector_store
        self.supabase = supabase

    async def sync_from_supabase(self):
        supabase_papers = []
        page_size = 1000
        start = 0
        while True:
            response = await self.supabase.table(settings.TABLE_NAME).select("*") \
                .range(start, start + page_size - 1).execute()
            print(response)
            if not response.data:
                break
            supabase_papers.extend(response.data)
            if len(response.data) < page_size:
                break
            start += page_size

        field_mapping = {
            "name": "title",
            "description": "abstract",
        }

        supabase_external_ids = set()

        for paper in supabase_papers:
            ext_id = paper["id"]
            supabase_external_ids.add(ext_id)

            clean_data = {}
            for src_field, dst_field in field_mapping.items():
                clean_data[dst_field] = paper.get(src_field)  # may be None if missing

            clean_data["keywords"] = paper.get("keywords", "")
            
            await self.paper_service.upsert_paper_by_external_id(ext_id, clean_data)

        local_ids = await self.paper_service.get_all_external_ids()
        to_delete = local_ids - supabase_external_ids
        for ext_id in to_delete:
            paper = await self.paper_service.get_paper_by_external_id(ext_id)
            if paper:
                await self.paper_service.delete_paper(paper.id)

        await self.vector_store.full_rebuild(self.paper_service)